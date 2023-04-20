import last10minutes

all_interact_summaries = last10minutes.getAllInteractSummary()
all_interact_summaries.to_csv('all_interact_summaries.csv', index=False)