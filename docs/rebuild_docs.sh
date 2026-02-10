rm -rf source/api
mkdir source/api

sphinx-apidoc \
  -o source/api \
  ../src/goripy \
  --force \
  --separate \
  --module-first
