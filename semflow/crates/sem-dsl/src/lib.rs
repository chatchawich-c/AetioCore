use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq)]
pub enum ParamSpec {
    Free,
    Fixed(f64),      // เช่น 1*y1, 0*y2
    Labeled(String), // เช่น b1*x1 (ผูก Equality Constraint)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    Sign(i8),
    Monotonic(bool),
    EdgeType(String), // linear, bspline, ispline, interact
    Custom(String, String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementKind {
    Reflective, // =~ (Mode A ใน PLS / Common Factor ใน CB-SEM)
    Formative,  // <~ (Mode B ใน PLS / Composite)
}

#[derive(Debug, Clone)]
pub struct Term {
    pub name: String,
    pub param: ParamSpec,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone)]
pub struct MeasurementBlock {
    pub latent: String,
    pub kind: MeasurementKind,
    pub terms: Vec<Term>,
}

#[derive(Debug, Clone)]
pub struct StructuralEdge {
    pub from: String,
    pub to: String,
    pub param: ParamSpec,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone)]
pub struct CovarianceEdge {
    pub lhs: String,
    pub rhs: String,
    pub param: ParamSpec,
}

/// Unified Intermediate Representation ใช้ร่วมกันทั้ง PLS, CB-SEM, LGM, HOC และ semflow-dl
#[derive(Debug, Clone)]
pub struct CausalGraphIR {
    pub latents: Vec<String>,
    pub observed: Vec<String>,
    pub higher_order_latents: Vec<String>,
    pub measurements: Vec<MeasurementBlock>,
    pub structural_edges: Vec<StructuralEdge>,
    pub covariances: Vec<CovarianceEdge>,
    pub topo_order: Vec<String>,
}

pub fn parse_formula(input: &str) -> Result<CausalGraphIR, String> {
    let mut measurements: Vec<MeasurementBlock> = Vec::new();
    let mut structural_edges: Vec<StructuralEdge> = Vec::new();
    let mut covariances: Vec<CovarianceEdge> = Vec::new();

    let mut latent_order: Vec<String> = Vec::new();
    let mut latent_set: HashSet<String> = HashSet::new();
    let mut all_referenced: Vec<String> = Vec::new();

    let push_unique = |vec: &mut Vec<String>, name: &str| {
        if !vec.iter().any(|x| x == name) {
            vec.push(name.to_string());
        }
    };

    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        if let Some((lhs, rhs)) = line.split_once("=~") {
            let latent = lhs.trim().to_string();
            if latent_set.insert(latent.clone()) {
                latent_order.push(latent.clone());
            }
            push_unique(&mut all_referenced, &latent);

            let terms = parse_rhs_terms(rhs, line_no + 1)?;
            for t in &terms {
                push_unique(&mut all_referenced, &t.name);
            }
            merge_measurement(&mut measurements, latent, MeasurementKind::Reflective, terms);
        } else if let Some((lhs, rhs)) = line.split_once("<~") {
            let latent = lhs.trim().to_string();
            if latent_set.insert(latent.clone()) {
                latent_order.push(latent.clone());
            }
            push_unique(&mut all_referenced, &latent);

            let terms = parse_rhs_terms(rhs, line_no + 1)?;
            for t in &terms {
                push_unique(&mut all_referenced, &t.name);
            }
            merge_measurement(&mut measurements, latent, MeasurementKind::Formative, terms);
        } else if let Some((lhs, rhs)) = line.split_once("~~") {
            let left = lhs.trim().to_string();
            push_unique(&mut all_referenced, &left);
            let terms = parse_rhs_terms(rhs, line_no + 1)?;
            for t in terms {
                push_unique(&mut all_referenced, &t.name);
                covariances.push(CovarianceEdge {
                    lhs: left.clone(),
                    rhs: t.name,
                    param: t.param,
                });
            }
        } else if let Some((lhs, rhs)) = line.split_once('~') {
            let target = lhs.trim().to_string();
            push_unique(&mut all_referenced, &target);
            let terms = parse_rhs_terms(rhs, line_no + 1)?;
            for t in terms {
                push_unique(&mut all_referenced, &t.name);
                structural_edges.push(StructuralEdge {
                    from: t.name,
                    to: target.clone(),
                    param: t.param,
                    constraints: t.constraints,
                });
            }
        } else {
            return Err(format!("Line {}: Syntax error near '{}'", line_no + 1, line));
        }
    }

    // แยกตัวแปรสังเกตได้ (Observed) ออกจากตัวแปรแฝง (Latent)
    // ช่วยให้รองรับ Higher-Order Constructs (Latent =~ Latent) และ MIMIC (Latent ~ Observed) อัตโนมัติ
    let observed: Vec<String> = all_referenced
        .iter()
        .filter(|v| !latent_set.contains(*v))
        .cloned()
        .collect();

    // ระบุ Higher-Order Latents (Latent ที่ตัวชี้วัดอย่างน้อย 1 ตัวเป็น Latent อื่น)
    let higher_order_latents: Vec<String> = measurements
        .iter()
        .filter(|m| m.terms.iter().any(|t| latent_set.contains(&t.name)))
        .map(|m| m.latent.clone())
        .collect();

    // ตรวจสอบ DAG Cycle บน Structural Edges ทั้งหมด
    let mut dag_nodes = latent_order.clone();
    for e in &structural_edges {
        push_unique(&mut dag_nodes, &e.from);
        push_unique(&mut dag_nodes, &e.to);
    }
    let topo_order = topological_sort(&dag_nodes, &structural_edges)?;

    Ok(CausalGraphIR {
        latents: latent_order,
        observed,
        higher_order_latents,
        measurements,
        structural_edges,
        covariances,
        topo_order,
    })
}

fn merge_measurement(
    blocks: &mut Vec<MeasurementBlock>,
    latent: String,
    kind: MeasurementKind,
    terms: Vec<Term>,
) {
    if let Some(existing) = blocks.iter_mut().find(|b| b.latent == latent) {
        existing.terms.extend(terms);
    } else {
        blocks.push(MeasurementBlock {
            latent,
            kind,
            terms,
        });
    }
}

fn parse_rhs_terms(rhs: &str, line_no: usize) -> Result<Vec<Term>, String> {
    let mut out = Vec::new();
    // แยกด้วย '+' เฉพาะนอกวงเล็บเหลี่ยม [...]
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, ch) in rhs.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            '+' if depth == 0 => {
                parts.push(rhs[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(rhs[start..].trim());

    for raw in parts {
        if raw.is_empty() {
            continue;
        }
        out.push(parse_single_term(raw, line_no)?);
    }
    Ok(out)
}

fn parse_single_term(raw: &str, line_no: usize) -> Result<Term, String> {
    let (head, constraints) = if let Some(open) = raw.find('[') {
        let close = raw
            .rfind(']')
            .ok_or_else(|| format!("Line {}: Unclosed '[' in '{}'", line_no, raw))?;
        let c_str = &raw[open + 1..close];
        (&raw[..open].trim(), parse_constraints(c_str, line_no)?)
    } else {
        (&raw.trim(), Vec::new())
    };

    let (param, name) = if let Some((prefix, var)) = head.split_once('*') {
        let p_str = prefix.trim();
        let v_str = var.trim().to_string();
        if let Ok(val) = p_str.parse::<f64>() {
            (ParamSpec::Fixed(val), v_str)
        } else {
            (ParamSpec::Labeled(p_str.to_string()), v_str)
        }
    } else {
        (ParamSpec::Free, head.to_string())
    };

    if name.is_empty() {
        return Err(format!("Line {}: Empty variable name in '{}'", line_no, raw));
    }

    Ok(Term {
        name,
        param,
        constraints,
    })
}

fn parse_constraints(inner: &str, line_no: usize) -> Result<Vec<Constraint>, String> {
    let mut out = Vec::new();
    for item in inner.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (k, v) = item
            .split_once('=')
            .ok_or_else(|| format!("Line {}: Expected key=value in '{}'", line_no, item))?;
        match (k.trim(), v.trim()) {
            ("sign", "+1" | "1") => out.push(Constraint::Sign(1)),
            ("sign", "-1") => out.push(Constraint::Sign(-1)),
            ("monotonic" | "mono", "true" | "True") => out.push(Constraint::Monotonic(true)),
            ("monotonic" | "mono", "false" | "False") => out.push(Constraint::Monotonic(false)),
            ("edge", val) => out.push(Constraint::EdgeType(val.to_string())),
            (key, val) => out.push(Constraint::Custom(key.to_string(), val.to_string())),
        }
    }
    Ok(out)
}

fn topological_sort(nodes: &[String], edges: &[StructuralEdge]) -> Result<Vec<String>, String> {
    let mut in_deg: HashMap<&str, usize> = nodes.iter().map(|n| (n.as_str(), 0)).collect();
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

    for e in edges {
        adj.entry(e.from.as_str()).or_default().push(e.to.as_str());
        *in_deg.entry(e.to.as_str()).or_insert(0) += 1;
    }

    let mut q: VecDeque<&str> = nodes
        .iter()
        .map(|s| s.as_str())
        .filter(|n| in_deg[n] == 0)
        .collect();

    let mut order = Vec::with_capacity(nodes.len());
    while let Some(u) = q.pop_front() {
        order.push(u.to_string());
        if let Some(nbrs) = adj.get(u) {
            for &v in nbrs {
                let d = in_deg.get_mut(v).unwrap();
                *d -= 1;
                if *d == 0 {
                    q.push_back(v);
                }
            }
        }
    }

    if order.len() != nodes.len() {
        return Err("Cycle detected in structural DAG".to_string());
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsl_fixed_labels_hoc_and_formative() {
        let spec = "
            F1 =~ 1*x1 + a*x2 + x3
            F2 <~ z1 + z2
            G  =~ F1 + F2
            Y  =~ 1*y1 + 0*y2
            Y  ~ b*G [sign=+1, edge=ispline]
            x1 ~~ x2
        ";
        let ir = parse_formula(spec).unwrap();
        assert_eq!(ir.latents, vec!["F1", "F2", "G", "Y"]);
        assert_eq!(ir.higher_order_latents, vec!["G"]);
        assert_eq!(ir.measurements[1].kind, MeasurementKind::Formative);
        assert_eq!(ir.measurements[0].terms[0].param, ParamSpec::Fixed(1.0));
        assert_eq!(
            ir.measurements[0].terms[1].param,
            ParamSpec::Labeled("a".into())
        );
        assert_eq!(ir.covariances.len(), 1);
    }
}
