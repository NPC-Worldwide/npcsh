import re

def is_email(s):
    pattern = r'^[\w\.\-]+@[^\w\.\-]+\.\w+$'
    return bool(re.match(pattern, s))

def is_url(s):
    pattern = r'^https?://\S+$'
    return bool(re.match(pattern, s))

if __name__ == "__main__":
    print(is_email("user@example.com"))
    print(is_url("https://example.com"))print(is_email("user-name@example.com"))print(is_url("http://example.com:8080"))