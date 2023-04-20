[event_info.json](event_info.json) contains a json/dictionary of the different events in mongo. It is organized like so:

```
{"eventCategory(category name)": {
    "first_seen": <date the eventCategory was first used>,
    "last_seen": <date the eventCategory was last used>,
    "count": <number of events with this eventCategory>,
    "eventAction(action_name)": {
        "first_seen": <date the eventAction was first used>,
        "last_seen": <date the eventAction was last used>,
        "count": <number of events with this eventAction>,
        "eventLabel(event label name)": {
            "first_seen": <date the eventLabel was first used>,
            "last_seen": <date the eventLabel was last used>,
            "count": <number of events with this eventLabel>,
            "field": {
                "first_seen": <date the field was first used>,
                "last_seen": <date the field was last used>,
                "count": <number of events with this field>,
            }
        }
    }
}}
```

This was created to help understand the events, how they have evolved and how to best handling cleaning/normalization of the old events.