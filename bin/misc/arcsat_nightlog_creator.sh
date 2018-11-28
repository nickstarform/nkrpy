#!/usr/bin/bash

OPT="${1}"

if [ "${OPT}" == '--help' ] || [ "${OPT}" == '-help' ] || [ "${OPT}" == 'help' ]; then
    echo "Either select a file or use '.' (without the ') to run on all files"
    echo "Attempts to auto generate a Focus Log (since it is a pain"
    exit
fi

if [ -z "${OPT}" ]; then
    OPT='.'
fi

if [ "${OPT}" == '.' ]; then
    echo 'Running Multifile'
    file=()
    for FILE in ./*.log; do
        echo "Running on file: ${FILE}"
        _tmp=''
        _tmp="${_tmp} $(grep 'True focal length' "${FILE}" | cut -d' ' -f1,8)"
        _tmp="${_tmp} $(grep 'True image center' "${FILE}" | cut -d' ' -f8,9,10,13,14,15)"
        _tmp="${_tmp} $(grep 'filter for pointing exposure' "${FILE}" | cut -d' ' -f8)"
        _tmp="${_tmp} $(grep 'avg FWHM' "${FILE}" | cut -d' ' -f7)"
        file+="${_tmp}"

    done
    echo ${focal[@]}
    echo ${center[@]}
    echo ${fil[@]}
    echo ${fwhm[@]}
else
    if [ ! -f "${OPT}" ]; then
        echo 'File not found.'
        exit
    fi
    FILE="${OPT}"
    echo "Running on file: ${FILE}"
    _tmp=''
    _tmp="${_tmp} $(grep 'True focal length' "${FILE}" | cut -d' ' -f1,8)"
    _tmp="${_tmp} $(grep 'True image center' "${FILE}" | cut -d' ' -f8,9,10,13,14,15)"
    _tmp="${_tmp} $(grep 'filter for pointing exposure' "${FILE}" | cut -d' ' -f8)"
    _tmp="${_tmp} $(grep 'avg FWHM' "${FILE}" | cut -d' ' -f7)"
    echo ${_tmp}
fi

# end of file
