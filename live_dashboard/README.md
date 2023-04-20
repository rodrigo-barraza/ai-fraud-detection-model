# Live Dashboard

This is an example dashboard for exploring analytics ideas that may be useful for einstein in inspecting user behaviour around purchases and trades.

**WARNING: This dashboard reads data from the MongoDB in real time and puts load on the server. Please confirm with the einstein team before running this to make sure it is ok to put load on the server or use a local copy instead.**


To run this make sure you have:

1. A credentials file with the connection string to a mongodb in the directory called `creds.json` containing a json like so:

```
{
    "connection_string": <mongo connection string>
}
```

2. Have installed the `einsteinds` package by going into its directory and running `pip install -e .` in bash.

3. Have installed the requirements using `pip install requirements.txt` and dealt with any install issues.

Then, 

In this directory in bash run `python dashboard.py` and follow the link given in the terminal window.