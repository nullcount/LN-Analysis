# Snapshot Time Machine

- Save space using diffs

## What's here?

`archive/` this folder keeps the earliest known snapshot `genesis` and the `latest` snapshot.

`diffs/` this folder keeps the diffs between snapshots taken

`snapshot` this script is your friend

## Usage

`./snapshot -s`

Just do a regular snapshot. Don't save diffs.

`./snapshot -d`

Make a diffed snapshot

`./snapshot -sd`

Make a diffed snapshot and a regular snapshot.

`./snapshot -l`

List availible snapshots to rebuild from diffs.

`./snapshot -r SNAPSHOT`

Recreate this snapshot from diffs. SNAPSHOT is an entry retruned from `snapshot -l`
