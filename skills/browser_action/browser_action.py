from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from npcpy.work.browser import get_current_driver

action = context.get('action', '').lower()
selector = context.get('selector', '')
value = context.get('value', '')

driver = get_current_driver()
if not driver:
    output = "No active browser. Use open_browser first."
    exit()

def find_element(sel):
    if sel.startswith('xpath:'):
        return driver.find_element(By.XPATH, sel[6:])
    else:
        return driver.find_element(By.CSS_SELECTOR, sel)

def wait_for_element(sel, timeout=10):
    if sel.startswith('xpath:'):
        return WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, sel[6:]))
        )
    else:
        return WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, sel))
        )

try:
    if action == 'click':
        elem = wait_for_element(selector)
        elem.click()
        output = f"Clicked: {selector}"

    elif action == 'type':
        elem = wait_for_element(selector)
        elem.click()  # Focus first
        elem.clear()
        elem.send_keys(value)
        output = f"Typed '{value}' into {selector}"

    elif action == 'set_value':
        # Force set value via JS, bypasses validation/calendars
        elem = wait_for_element(selector)
        driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input', {bubbles: true})); arguments[0].dispatchEvent(new Event('change', {bubbles: true}));", elem, value)
        output = f"Set value '{value}' on {selector}"

    elif action == 'type_and_enter':
        elem = wait_for_element(selector)
        elem.clear()
        elem.send_keys(value)
        elem.send_keys(Keys.RETURN)
        output = f"Typed '{value}' and pressed Enter"

    elif action == 'select':
        elem = wait_for_element(selector)
        select = Select(elem)
        select.select_by_visible_text(value)
        output = f"Selected '{value}' in {selector}"

    elif action == 'wait':
        timeout = int(value) if value else 10
        wait_for_element(selector, timeout)
        output = f"Element found: {selector}"

    elif action == 'scroll':
        if value == 'down':
            driver.execute_script("window.scrollBy(0, 500)")
        elif value == 'up':
            driver.execute_script("window.scrollBy(0, -500)")
        elif selector:
            elem = find_element(selector)
            driver.execute_script("arguments[0].scrollIntoView();", elem)
        output = f"Scrolled {value or 'to element'}"

    elif action == 'get_text':
        elem = wait_for_element(selector)
        output = f"Text: {elem.text}"

    elif action == 'get_page':
        title = driver.title
        url = driver.current_url
        body = driver.find_element(By.TAG_NAME, 'body')
        text = body.text[:3000]
        output = f"Page: {title} ({url})\n\nContent:\n{text}"

    elif action == 'get_elements':
        elements = []

        def is_visible(el):
            try:
                return el.is_displayed() and el.size['width'] > 0
            except:
                return False

        def safe_selector(tag, el):
            eid = el.get_attribute('id')
            name = el.get_attribute('name')
            if eid and '.' not in eid and ' ' not in eid:
                return '#' + eid
            elif eid:
                return f'{tag}[id="{eid}"]'
            elif name:
                return f'{tag}[name="{name}"]'
            return None

        # Get inputs
        for inp in driver.find_elements(By.CSS_SELECTOR, 'input:not([type="hidden"])'):
            if not is_visible(inp):
                continue
            sel = safe_selector('input', inp)
            if not sel:
                ph = inp.get_attribute('placeholder')
                if ph:
                    sel = f'input[placeholder="{ph}"]'
                else:
                    continue
            info = {'tag': 'input', 'type': inp.get_attribute('type') or 'text', 'selector': sel}
            info['placeholder'] = inp.get_attribute('placeholder') or ''
            elements.append(info)

        # Get buttons
        for btn in driver.find_elements(By.CSS_SELECTOR, 'button, input[type="submit"], input[type="button"]'):
            if not is_visible(btn):
                continue
            sel = safe_selector('button', btn)
            if not sel and btn.text:
                sel = f'xpath://button[contains(text(),"{btn.text[:30]}")]'
            if not sel:
                continue
            info = {'tag': 'button', 'selector': sel, 'text': (btn.text or '')[:50]}
            elements.append(info)

        # Get select dropdowns
        for s in driver.find_elements(By.TAG_NAME, 'select'):
            if not is_visible(s):
                continue
            sel = safe_selector('select', s)
            if not sel:
                continue
            opts = [o.text for o in s.find_elements(By.TAG_NAME, 'option')[:5]]
            info = {'tag': 'select', 'selector': sel, 'options': opts}
            elements.append(info)

        # Get links
        for link in driver.find_elements(By.TAG_NAME, 'a')[:30]:
            if not is_visible(link) or not link.text or len(link.text) < 2:
                continue
            sel = safe_selector('a', link)
            if not sel:
                sel = f'xpath://a[contains(text(),"{link.text[:30]}")]'
            info = {'tag': 'a', 'selector': sel, 'text': link.text[:50]}
            elements.append(info)

        output = f"Found {len(elements)} visible elements:\n"
        for el in elements[:40]:
            output += f"{el['tag']}: {el.get('selector', '')} "
            if el.get('text'):
                output += f'"{el["text"][:30]}" '
            if el.get('placeholder'):
                output += f'placeholder="{el["placeholder"]}" '
            if el.get('options'):
                output += f"opts={el['options'][:3]} "
            output += "\n"

    elif action == 'press_key':
        key_map = {
            'enter': Keys.RETURN, 'return': Keys.RETURN,
            'tab': Keys.TAB,
            'escape': Keys.ESCAPE, 'esc': Keys.ESCAPE,
            'down': Keys.DOWN, 'up': Keys.UP,
            'left': Keys.LEFT, 'right': Keys.RIGHT,
            'backspace': Keys.BACKSPACE,
            'delete': Keys.DELETE,
        }
        key = key_map.get(value.lower(), value)
        if selector:
            elem = find_element(selector)
            elem.send_keys(key)
        else:
            driver.find_element(By.TAG_NAME, 'body').send_keys(key)
        output = f"Pressed key: {value}"

    else:
        output = f"Unknown action: {action}"

except Exception as e:
    output = f"Browser action failed: {str(e)}"
