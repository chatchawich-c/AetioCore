#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-dist/r-pkg}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp -R r-pkg/. "$OUT_DIR/"

# คัดลอก Pure Rust Core Crates เข้าไปเป็น Self-Contained ภายในแพ็กเกจ R
mkdir -p "$OUT_DIR/src/rust/crates"
for c in sem-traits sem-dsl sem-autodiff sem-pls sem-cb; do
  cp -R "crates/$c" "$OUT_DIR/src/rust/crates/$c"
  # แทนที่ workspace = true ในแต่ละ crate เพื่อให้ build อิสระนอก root workspace ได้
  python3 - "$OUT_DIR/src/rust/crates/$c/Cargo.toml" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
txt = p.read_text()
txt = re.sub(r'version\.workspace\s*=\s*true', 'version = "0.2.0"', txt)
txt = re.sub(r'edition\.workspace\s*=\s*true', 'edition = "2021"', txt)
txt = re.sub(r'faer\.workspace\s*=\s*true', 'faer = "0.20"', txt)
txt = re.sub(r'rayon\.workspace\s*=\s*true', 'rayon = "1.10"', txt)
p.write_text(txt)
PY
done

# อัปเดต path ใน r-pkg/src/rust/Cargo.toml จาก ../../../crates/ เป็น ./crates/
python3 - "$OUT_DIR/src/rust/Cargo.toml" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
p.write_text(p.read_text().replace("../../../crates/", "./crates/"))
PY

# ลบ target ชั่วคราวหากมีติดมา
rm -rf "$OUT_DIR/src/rust/target"
echo "Self-contained R package assembled at: $OUT_DIR"
