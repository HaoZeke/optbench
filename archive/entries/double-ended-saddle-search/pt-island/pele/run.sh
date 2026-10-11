#!/bin/bash
tar -xzf pt-island-con.tgz
python dec_morse.py
python getdata.py
rm -r pt-island-con

echo "code_file pele-$(python get_pele_version.py).tgz" >> benchmark.dat
