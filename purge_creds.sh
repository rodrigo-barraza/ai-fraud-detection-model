git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch creds\production_creds.json' \
--prune-empty --tag-name-filter cat -- --all