# Automated Channel Backups with Raspiblitz

Raspiblitz supports automated channel backups to external host.

1. Every 1min the `_background.sh` script runs as a service to check if the channel backup file has changed.

2. If so, it creates a copy to `/home/admin/.lnd/data/chain/[..]/channel.backup`

3. Then the script checks for additional offsite targets set in `/mnt/hdd/raspiblitz.conf`

## Setup Using Main Menu

From the main menu, go to Settings, then enable static channel backups.

## Manual Config

### SCP Backup Target

In the `raspiblitz.conf` the parameter `scpBackupTarget` can be set with the value formatted like `[USER]@[SERVER]:[DIRPATH-WITHOUT-ENDING-/]`. On that remote server the publickey of the RaspiBlitz root user needs to be part of the authorized keys - so that no password is needed for the background script to make the backup.

The script `/home/admin/config.scripts/internet.sshpubkey.sh` helps on init, show and transfer ssh-pubkey to a remote server.

### Dropbox Backup Target

In the raspiblitz.conf the parameter `dropboxBackupTarget` can be set to a DropBox Authtoken. See how to get that token here: https://gist.github.com/vindard/e0cd3d41bb403a823f3b5002488e3f90
