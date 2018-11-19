#!/usr/bin/bash

source='/home/reynolds/github/nickalaskreynolds/nkrpy/nkrpy/apo/combined_orders_template.ipynb'
dest=($(echo "${source}" | tr "/" "\n"))
len="${#dest[@]}"
let 'len= len - 1'
dest="${dest[${len}]}"

files=(./*tellcor.fits)
for f in "${files[@]}"; do
    if [ -f "${f}" ]; then
        name=($(echo "${f}" | tr "." "\n"))
        fin="./${dest%"template.ipynb"}${name[0]#"/"}.ipynb"
        if [ ! -f "${fin}" ]; then
            cp -n "${source}" "${fin}"
        else
            echo "Won't overwrite ${fin}"
        fi
    fi
done

ls *.ipynb

# end of file
