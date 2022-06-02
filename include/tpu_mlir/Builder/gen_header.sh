#!/bin/bash
MODEL_FBS_HEADER="$2_fbs.h"
MODEL_FBS="$1"
function generate_fbs()
{
echo "const char * schema_text =" > $MODEL_FBS_HEADER
while read line; do
    echo "\"$line\\n\"">> $MODEL_FBS_HEADER
done < $MODEL_FBS
echo "\"\";" >> $MODEL_FBS_HEADER
}

generate_fbs


