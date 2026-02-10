rm -rf api
mkdir api

sphinx-apidoc \
  -o api \
  ../src/goripy \
  --force \
  --separate \
  --module-first
