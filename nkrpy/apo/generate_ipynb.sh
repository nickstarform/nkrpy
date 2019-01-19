#!/usr/bin/bash

# generic multi-order
source='/home/reynolds/github/nickalaskreynolds/nkrpy/nkrpy/apo/combined_orders_template.ipynb'
dest=($(echo "${source}" | tr "/" "\n"))
let 'len= len - 1'
dest="${dest[${len}]}"
fin_dest="${dest}"
len="${#dest[@]}"

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

# generic single_order
source='/home/reynolds/github/nickalaskreynolds/nkrpy/nkrpy/apo/single_order_template.ipynb'
cp -n "${source}" "./"

ls *.ipynb

# end of file
