#!/bin/bash
TIMESTAMP=$(date +%Y-%m-%d%t%H:%M)
MESSAGE="$*"
if [ "" == "${MESSAGE}" ];
then
  echo commit message is required!
  exit
  MESSAGE="(no message)"
fi
git add -A
git commit -m "${TIMESTAMP}: ${MESSAGE};"
