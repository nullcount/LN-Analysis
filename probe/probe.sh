#!/bin/bash

echo Node to send to:
read reciever
echo Channel peer for first hop: 
read out

bos probe $reciever --out $out --find-max
