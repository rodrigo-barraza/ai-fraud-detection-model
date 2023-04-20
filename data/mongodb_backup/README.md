# MongoDB Backup

To avoid putting huge loads on the production server, we want to take a periodic backup of the production server and use a local copy when building and testing machine learning algorithms. To accomplish this you will need to:

- Install MongoDB locally
- Take a full backup of the production mongo database
- Restore a copy of the production database to a local server

### Installing Mongo Locally:

Follow the instructions to [install MongoDB Community Edition](https://docs.mongodb.com/manual/administration/install-community/) locally.

### Backing up the Production Database:

1. Make sure you have the username and password as environmental variables in your bash terminal with the following commands:

`export MONGOUSER=<username>`
`export MONGOPASSWORD=<password>`

Note that this works with the read-only mongo password, so this is what you should use.

2. Run `bash backup.sh` to download the production database. **WARNING: This generates load on the production server so confirm this is ok before proceeding.**
3. Start the local mongo server with `sudo service mongod start`
4. Restore the production database to the local server with `bash restore_to_local.sh`