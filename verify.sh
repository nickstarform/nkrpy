#/bin/bash
version=$(cat .version)
if [ "${version}" == "${CI_COMMIT_REF_NAME}" ]; then
    exit 0
else
    exit 255
fi