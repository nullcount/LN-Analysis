# Using Testnet on Raspiblitz

swich the node to testnet

`cd ~/config.scripts/`

`bash network.chain.sh testnet`

This will set LND and Bitcoin Core to use testnet.

There are now 2 new folders:

Chain and wallet data - `/mnt/hdd/bitcoin/testnet3`

Wallet, macroons, and backups - `/mnt/hdd/lnd/data/chain/bitcoin/testnet`

Check the bitcoind logs and lnd logs for errors.

## Service Quirks

Other services may need additional config to use testnet

### JoinMarket/JoinBox

Using the JoinBox Menu, 

1. `CONFIG`
2. `JMCONF` - Edit the config manually
3. set `network = testnet`
4. set `rpc_port = 18332`
5. `TAB + ENTER` to select `< OK >` and save
6. Create new wallet normally

