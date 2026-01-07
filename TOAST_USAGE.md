# Toast Notification System

A comprehensive toast notification system that displays messages for 5 seconds by default with full customization support.

## Features

- **5-second auto-dismiss** - Toasts automatically disappear after 5 seconds
- **Manual dismiss** - Users can click the X button to close immediately
- **4 Types** - Success (green), Error (red), Warning (yellow), Info (blue)
- **Icons** - Each type has a contextual icon
- **Smooth animations** - Fade in/out transitions
- **Stacked display** - Multiple toasts stack vertically
- **Fully customizable** - Duration, type, message all configurable
- **Backend integration** - Send toasts from FastAPI endpoints via response headers
- **Frontend integration** - JavaScript API and custom events

## Usage

### Frontend - JavaScript

#### Basic Usage

```javascript
// Success toast
showToast('Order placed successfully!', 'success');

// Error toast
showToast('Failed to place order', 'error');

// Warning toast
showToast('Low balance detected', 'warning');

// Info toast
showToast('Market is closed on weekends', 'info');
```

#### Using ToastManager

```javascript
// Using the global Toast object
Toast.success('Trade executed successfully!');
Toast.error('Connection lost');
Toast.warning('You have unsaved changes');
Toast.info('New signal detected');
```

#### Custom Duration

```javascript
// Display for 10 seconds
showToast('Custom message', 'success', 10000);

// Display for 2 seconds
Toast.info('Quick notification', 2000);
```

### Frontend - HTML & Buttons

```html
<!-- Simple button with click handler -->
<button onclick="Toast.success('Button clicked!')">Click Me</button>

<!-- With custom duration -->
<button onclick="Toast.warning('Action will be irreversible', 'warning', 3000)">Delete</button>
```

### Frontend - HTML & HTMX

```html
<!-- HTMX will automatically show toast from server headers -->
<button
    hx-post="/api/place-order"
    hx-target="#result">
    Place Order
</button>
```

When the endpoint returns response headers:
```
X-Toast-Message: Order placed successfully!
X-Toast-Type: success
X-Toast-Duration: 5000
```

The toast will automatically appear.

### Frontend - Custom Events

```javascript
// Dispatch a custom toast event
const event = new CustomEvent('showToast', {
    detail: {
        message: 'Custom event toast',
        type: 'success',
        duration: 5000
    }
});
document.dispatchEvent(event);
```

---

## Backend Integration

### Python - FastAPI Endpoints

#### Basic Usage with ToastResponse

```python
from fastapi import APIRouter
from app.core.toast import ToastResponse

router = APIRouter()

@router.post("/place-order")
async def place_order(request: Request):
    # Process order...
    return ToastResponse(
        "<h1>Success</h1>",
        toast_message="Order placed successfully!",
        toast_type="success"
    )
```

#### With HTMLToastResponse

```python
from fastapi.templating import Jinja2Templates
from app.core.toast import HTMLToastResponse

templates = Jinja2Templates(directory="templates")

@router.post("/trade")
async def execute_trade(request: Request):
    # Execute trade...
    html = templates.TemplateResponse(
        "trade_result.html",
        {"request": request, "trade_id": "12345"}
    )

    return HTMLToastResponse(
        str(html.body),
        toast_message="Trade executed: Bought 50 shares",
        toast_type="success",
        toast_duration=5000
    )
```

#### Adding Toast to Existing Responses

```python
from app.core.toast import add_toast_to_response

@router.get("/download")
async def download_file():
    response = FileResponse("file.pdf")
    return add_toast_to_response(
        response,
        message="File downloading...",
        toast_type="info"
    )
```

#### Custom Duration from Backend

```python
@router.post("/long-running-operation")
async def long_operation(request: Request):
    # Some operation...
    return ToastResponse(
        "Done",
        toast_message="This operation took a while",
        toast_type="info",
        toast_duration=8000  # Show for 8 seconds
    )
```

### Response Header Format

When sending toasts from the backend, use these headers:

```
X-Toast-Message: Your message here
X-Toast-Type: success|error|warning|info
X-Toast-Duration: 5000  # milliseconds
```

---

## Toast Types Reference

| Type | Color | Icon | Use Case |
|------|-------|------|----------|
| **success** | Green (Emerald) | ✓ Check Circle | Successful operations |
| **error** | Red (Rose) | ⓘ Alert Circle | Failed operations, errors |
| **warning** | Yellow (Amber) | ⚠ Alert Triangle | Warnings, cautions |
| **info** | Blue | ⓘ Info | General information |

---

## Examples

### Example 1: Place Trade Order

```html
<!-- Frontend -->
<button
    hx-post="/api/trade/place"
    hx-target="#orders">
    Place Order
</button>
```

```python
# Backend
from app.core.toast import ToastResponse

@router.post("/api/trade/place")
async def place_trade(request: Request):
    # Validate & place order
    if not validate_order():
        return ToastResponse(
            "Error",
            status_code=400,
            toast_message="Invalid order parameters",
            toast_type="error"
        )

    order = create_order()

    return ToastResponse(
        "OK",
        toast_message=f"Order #{order.id} placed for {order.symbol}",
        toast_type="success"
    )
```

### Example 2: Form Submission

```html
<!-- Frontend -->
<form hx-post="/api/settings/save">
    <input name="risk_level" type="number" />
    <button type="submit">Save Settings</button>
</form>
```

```python
# Backend
@router.post("/api/settings/save")
async def save_settings(request: Request):
    data = await request.form()

    try:
        update_user_settings(data)
        return ToastResponse(
            "Settings saved",
            toast_message="Your settings have been updated",
            toast_type="success"
        )
    except Exception as e:
        return ToastResponse(
            f"Error: {str(e)}",
            status_code=400,
            toast_message=f"Failed to save: {str(e)}",
            toast_type="error"
        )
```

### Example 3: Manual Trigger

```javascript
// User performs action
document.getElementById('connectBtn').addEventListener('click', async () => {
    try {
        await connectToExchange();
        Toast.success('Connected to exchange!');
    } catch (error) {
        Toast.error('Connection failed: ' + error.message);
    }
});
```

### Example 4: Timed Operations

```javascript
// Show warning for time-sensitive operations
function placeTimedTrade() {
    Toast.warning('Trade will expire in 60 seconds', 'warning', 60000);

    setTimeout(() => {
        // Execute trade
        Toast.success('Trade executed!');
    }, 55000);
}
```

---

## Styling

Toasts use Tailwind CSS classes and are fully responsive. The default container is:

- **Position**: Bottom-right corner
- **Spacing**: 1rem (16px) from edges
- **Z-index**: 50 (above most content)
- **Width**: Auto-fit to content, max reasonable width

To customize the container, edit `base.html`:

```html
<!-- Toast Container -->
<div id="toast-container" class="fixed bottom-4 right-4 z-50 flex flex-col gap-2"></div>
```

Change positioning:
- Bottom-left: `bottom-4 left-4`
- Top-right: `top-4 right-4`
- Top-left: `top-4 left-4`

---

## Tips & Best Practices

1. **Keep messages concise** - Toast messages should be short (under 50 chars ideal)
2. **Use appropriate types** - Match the message tone to the type
3. **Don't overuse** - Save toasts for important feedback, not every action
4. **Consistent duration** - Stick to 5 seconds unless there's a reason not to
5. **Test accessibility** - Ensure toast messages are clear and readable
6. **Error messages** - Be specific about what went wrong
7. **Success messages** - Be encouraging and positive
8. **Combine with other UI** - Toasts work best alongside form validation, loading states, etc.

---

## Troubleshooting

### Toast not appearing?
1. Check browser console for errors
2. Verify `X-Toast-Message` header is present in response
3. Ensure toast container exists in HTML
4. Check z-index conflicts with other elements

### Text too long?
- Keep messages under 60 characters
- Use line breaks by limiting width
- Consider showing in a modal for longer messages

### Not auto-dismissing?
- Check toast duration is set correctly
- Verify JavaScript is not blocked
- Check browser console for errors

---

## API Reference

### showToast(message, type, duration)

```javascript
/**
 * Show a toast notification
 * @param {string} message - The message to display
 * @param {string} type - 'success', 'error', 'warning', or 'info'
 * @param {number} duration - Display duration in milliseconds (default: 5000)
 */
showToast(message, type, duration);
```

### Toast.* Methods

```javascript
Toast.success(message, duration);    // Green success toast
Toast.error(message, duration);      // Red error toast
Toast.warning(message, duration);    // Yellow warning toast
Toast.info(message, duration);       // Blue info toast
Toast.show(message, type, duration); // Custom toast
```

### Python Classes

```python
ToastResponse(content, status_code, headers, toast_message, toast_type, toast_duration)
HTMLToastResponse(content, status_code, headers, toast_message, toast_type, toast_duration)
JSONToastResponse(content, status_code, headers, toast_message, toast_type, toast_duration)
add_toast_to_response(response, message, toast_type, duration)
```
