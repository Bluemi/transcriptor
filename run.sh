#!/bin/bash

case "$1" in
	r)
		shift
		python3 transcriptor/main.py
		;;
	*)
		shift
		echo "invalid option: $@"
esac
