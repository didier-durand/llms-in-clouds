#!/bin/bash
echo "adapting OpenManus setup..."

# fail at first command with RC != 0
set -e


FILE="requirements.txt"
# replacement
sed "s/~=/>=/g" -i $FILE
# check if ok
test "$(grep -c "~="  $FILE)" -eq 0