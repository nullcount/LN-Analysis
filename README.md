# LN-Analysis with Python

## Getting Started

1. `git clone https://github.com/count-null/LN-Analysis`
2. `cd LN-Analysis`
3. `python3 -m venv venv`
4. `source venv/bin/activate`
5. `pip3 install -r requirements.txt`
6. `cp example.config.yml config.yml`
7. Edit `config.yml` if desired

### Aquire a Network Snapshot

You will need a snapshot of the Lightning Network graph. This project uses a special format with a timestamp.

i.e.

```
{
  "timestamp": 1608063155,
  "graph": <normal output of lncli describegraph>
}
```

If you are using your own dump, modify it to include the `timestamp` and `graph` entry. Otherwise, you can use `snapshot.py` to create a snapshot on your own lightning node.

To create a snapshot locally (on the node)
`python3 snapshot.py`

To upload a snapshot to another host
`python3 snapshot.py --remote-write`

Use with [cron](https://www.howtogeek.com/101288/how-to-schedule-tasks-on-linux-an-introduction-to-crontab-files/) to get up-to-date snapshots.

NOTE: Each subsequent snapshot will be overwritten.
TODO: Implement a method for collecting historical data.

### Make Graph

Run `graphize.py /path/to/graph.json` to greate a [NetworkX](https://networkx.org/) graph from the snapshot. This will add a new [pickle](https://wiki.python.org/moin/UsingPickle) to your `jar` directory.

### Do Some Analysis!

In `example_script.py` we show how to load in a pickle for analysis.
The `picklePicker` function will render a menu of saved pickles and allow the user to select which graph to use for this script.
