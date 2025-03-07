#!/bin/bash
echo "patching SGLang setup..."

# fail at first command with RC != 0
set -e

# extend keep alive timeout to 500s
FILE='/usr/local/lib/python3.12/site-packages/sglang/srt/entrypoints/http_server.py'
# to check presence of 5 and fail if not present
grep 'timeout_keep_alive=5,' $FILE
# replacement
sed 's/timeout_keep_alive=5,/timeout_keep_alive=500,/g' -i $FILE
# to obtain RC=1 and fail if change not made
grep 'timeout_keep_alive=500,' $FILE

##fix code issue in scheduler.py for 0.4.3.post3
#FILE='/usr/local/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py'
## to check presence incorrect code and fail if not present
#grep 'self.token_to_kv_pool.available_size()' $FILE
## replacement
#sed 's/self.token_to_kv_pool.available_size()/self.token_to_kv_pool_allocator.available_size()/g' -i $FILE
## to obtain RC=1 and fail if change not made
#grep 'self.token_to_kv_pool_allocator.available_size()' $FILE