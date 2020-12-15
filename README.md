# LN-Analysis with Python

## Getting Started

You will need a snapshot of the Lightning Network graph. This project uses a special dump with a timestamp.

i.e.

```
{
  "timestamp": 1608063155,
  "graph": <normal output of lncli describegraph>
}
```

If you are using your own dump, modify it to include the `timestamp` and `graph` entry. Otherwise, you can use `snapshot.py` to create your own snapshot on your own lightning node.
