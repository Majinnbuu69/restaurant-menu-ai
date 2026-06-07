import server
html = server.dashboard_html(2000)
lines = html.split('\n')
script_start = next((i for i, l in enumerate(lines) if '<script>' in l), 0)
print(f"Script starts at line {script_start + 1}")
for i in range(script_start, len(lines)):
    line = lines[i]
    # A JS string split across lines: line ends with ' or " but the string opened on this line
    # Simple heuristic: look for join( followed by end of line with quote
    if line.rstrip().endswith("'") and i + 1 < len(lines) and lines[i+1].strip().startswith("'"):
        print(f"BROKEN STRING at L{i+1}: {repr(line[-80:])}")
        print(f"  continues at L{i+2}: {repr(lines[i+1][:80])}")
print("Done")
