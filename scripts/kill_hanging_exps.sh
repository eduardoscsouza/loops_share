#!/bin/bash

ps -u "$(whoami)" -opid,cmd | egrep "[0-9]+ python .*\.py$" -o | egrep "^[0-9]+" -o | xargs -I{} kill {}
