# LN Graph Snapshot Methods

Provides different methods for taking snapshots of the graph

## Methods

#### Python Method

Using python to execute lncli describegraph in shell. Then write to file. Offers methods to upload graph offsite.

#### Diff Method

Using bash script to keep diffs between snapshots and rebuild snapshots from applying patches sequentially.

#### Tar Method

Simple bash script to take full snapshots then tar archive them at some threshold. This method is most space efficient. 
