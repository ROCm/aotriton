GIT_ORIGIN=$(git remote get-url origin)
if [[ ${GIT_ORIGIN} == "https://"* ]]; then
  GIT_HTTPS_ORIGIN=${GIT_ORIGIN}
else
  git_path=$(echo "${GIT_ORIGIN}"|cut -d ":" -f 2)
  GIT_HTTPS_ORIGIN="https://github.com/${git_path}"
fi
