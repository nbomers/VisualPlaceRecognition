import re
# Trailer und Sitzungszeilen weg, und die eine Erwaehnung im Text.
message = re.sub(rb"\n?Co-Authored-By: Claude[^\n]*", b"", message)
message = re.sub(rb"\n?Claude-Session:[^\n]*", b"", message)
message = message.replace(b" by claude and some small changes", b" and some small changes")
return message.rstrip(b"\n") + b"\n"
