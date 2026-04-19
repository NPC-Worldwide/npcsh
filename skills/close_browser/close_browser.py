from npcpy.work.browser import close_current

if close_current():
    output = "Browser closed."
else:
    output = "No active browser session."
