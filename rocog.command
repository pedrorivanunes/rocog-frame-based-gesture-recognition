#!/usr/bin/env bash
# Double-click this in the Finder to open the menu in a Terminal window. The
# window stays open at the end, so that the report can be read.
cd "$(dirname "$0")" || exit 1
./rocog
echo
read -r -p "Press Enter to close this window. " _
