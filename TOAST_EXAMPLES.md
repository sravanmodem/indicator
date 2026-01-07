# Toast Notification - Quick Examples

## Quick Reference

### Show a Toast (Frontend)

```javascript
// These are all you need to remember!
showToast('Message', 'success');    // 5 sec auto-dismiss
showToast('Message', 'error');
showToast('Message', 'warning');
showToast('Message', 'info');

// Or using the Toast object
Toast.success('Message');
Toast.error('Message');
Toast.warning('Message');
Toast.info('Message');
```

---

## Real-World Examples

### 1. Paper Trading - Place Order

```html
<!-- HTML -->
<button hx-post="/api/paper/place-order" hx-target="#orders">
    Place Order
</button>
```

```python
# Backend (app/api/paper_trading.py)
from app.core.toast import ToastResponse

@router.post("/place-order")
async def place_order(request: Request):
    try:
        order = await create_paper_order(request)

        return ToastResponse(
            f"Order {order.id}",
            toast_message=f"Order placed: {order.symbol} @ {order.price}",
            toast_type="success"
        )
    except Exception as e:
        return ToastResponse(
            "Error",
            status_code=400,
            toast_message=f"Failed to place order: {str(e)}",
            toast_type="error"
        )
```

### 2. Settings Update

```html
<!-- HTML Form -->
<form hx-post="/api/settings/update">
    <input name="risk_level" type="number" value="2" />
    <button type="submit">Save Settings</button>
</form>
```

```python
# Backend
@router.post("/settings/update")
async def update_settings(request: Request):
    data = await request.form()
    update_user_settings(data)

    return ToastResponse(
        "OK",
        toast_message="Settings updated successfully",
        toast_type="success"
    )
```

### 3. Zerodha Authentication

```python
# app/api/auth.py
from app.core.toast import ToastResponse

@router.get("/login/callback")
async def login_callback(request_id: str, request: Request):
    try:
        auth_service = get_auth_service()
        auth_service.process_callback(request_id)

        return ToastResponse(
            "Authenticated",
            toast_message="Successfully logged in to Zerodha",
            toast_type="success"
        )
    except Exception as e:
        return ToastResponse(
            "Auth Failed",
            status_code=400,
            toast_message="Authentication failed. Please try again.",
            toast_type="error"
        )
```

### 4. Trade Execution

```javascript
// Frontend - Manual trigger
document.getElementById('executeBtn').addEventListener('click', async () => {
    try {
        const response = await fetch('/api/execute-trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: 'NIFTY', strike: 23000 })
        });

        if (response.ok) {
            const data = await response.json();
            Toast.success(`Trade executed: ${data.symbol}`);
        } else {
            Toast.error('Trade execution failed');
        }
    } catch (error) {
        Toast.error('Network error: ' + error.message);
    }
});
```

### 5. Bulk Operations with Different Durations

```javascript
// Show different toasts with custom durations
Toast.warning('Processing 100 orders...', 3000);

// After 5 seconds
setTimeout(() => {
    Toast.info('50 orders processed', 2000);
}, 5000);

// After 10 seconds
setTimeout(() => {
    Toast.success('All orders processed!', 5000);
}, 10000);
```

### 6. Form Validation Feedback

```html
<!-- HTML -->
<form id="tradeForm">
    <input name="quantity" type="number" required />
    <input name="price" type="number" required />
    <button type="submit">Execute Trade</button>
</form>

<script>
document.getElementById('tradeForm').addEventListener('submit', (e) => {
    e.preventDefault();

    const quantity = document.querySelector('[name="quantity"]').value;
    const price = document.querySelector('[name="price"]').value;

    if (quantity <= 0) {
        Toast.warning('Quantity must be greater than 0');
        return;
    }

    if (price <= 0) {
        Toast.warning('Price must be greater than 0');
        return;
    }

    // Execute trade
    Toast.success(`Executing: ${quantity} @ ${price}`);
});
</script>
```

### 7. WebSocket Updates

```javascript
// In your WebSocket message handler
ws.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'signal') {
        Toast.info(`New ${data.signal} signal for ${data.symbol}`);
    } else if (data.type === 'price_alert') {
        Toast.warning(`Price alert: ${data.symbol} at ${data.price}`);
    }
});
```

### 8. API Error Handling

```javascript
// Generic API fetch with toast error handling
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            const error = await response.json();
            Toast.error(error.detail || 'Request failed');
            return null;
        }

        return await response.json();
    } catch (error) {
        Toast.error('Network error: ' + error.message);
        return null;
    }
}

// Usage
const result = await apiCall('/api/get-data');
```

### 9. Long-running Operations

```python
# Backend - Long operation with extended toast duration
@router.post("/process-large-file")
async def process_file(request: Request):
    file = await request.form()

    # Process file (takes time)
    result = await heavy_processing(file)

    return ToastResponse(
        "Done",
        toast_message="Large file processing completed successfully",
        toast_type="success",
        toast_duration=7000  # Show for 7 seconds instead of 5
    )
```

### 10. Conditional Toasts Based on Status

```python
# Backend - Different messages for different scenarios
@router.post("/place-trade")
async def place_trade(request: Request):
    order_data = await request.json()

    # Check various conditions
    if not is_market_open():
        return ToastResponse(
            "After Hours",
            toast_message="Market is closed. Order will be placed at market open.",
            toast_type="warning"
        )

    if portfolio_margin_insufficient():
        return ToastResponse(
            "Insufficient Funds",
            status_code=400,
            toast_message="You don't have enough margin for this trade",
            toast_type="error"
        )

    # Place order
    order = create_order(order_data)

    if order.filled_quantity == 0:
        return ToastResponse(
            "Partial",
            toast_message=f"Order placed but not filled yet",
            toast_type="info"
        )

    return ToastResponse(
        "Success",
        toast_message=f"Order executed: {order.symbol} {order.quantity} @ {order.price}",
        toast_type="success"
    )
```

---

## Common Patterns

### Pattern 1: Success After Async Operation

```javascript
async function deleteOrder(orderId) {
    try {
        const response = await fetch(`/api/orders/${orderId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            Toast.success('Order cancelled successfully');
            // Refresh list
            location.reload();
        }
    } catch (error) {
        Toast.error('Failed to cancel order');
    }
}
```

### Pattern 2: Multiple Step Process

```javascript
async function executeComplexTrade() {
    Toast.info('Step 1: Validating trade...', 2000);

    // Step 1
    if (!validateTrade()) {
        Toast.error('Trade validation failed');
        return;
    }

    await sleep(2000);
    Toast.info('Step 2: Checking margin...', 2000);

    // Step 2
    if (!hasMargin()) {
        Toast.error('Insufficient margin');
        return;
    }

    await sleep(2000);
    Toast.info('Step 3: Executing...', 2000);

    // Step 3
    const result = await executeTrade();

    if (result.success) {
        Toast.success('Trade executed successfully!');
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
```

### Pattern 3: Error Recovery

```javascript
async function withRetry(fn, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === maxRetries - 1) {
                Toast.error(`Failed after ${maxRetries} attempts`);
                throw error;
            }
            Toast.warning(`Attempt ${i + 1} failed, retrying...`);
            await sleep(1000);
        }
    }
}
```

---

## Styling Customization

If you want to customize toast appearance, edit `base.html` in the toast styling section:

```javascript
// In base.html, modify the styles object:
const styles = {
    success: {
        color: 'bg-green-600 border-green-500',      // Change these
        icon: 'check-circle',
        text: 'text-green-50'
    },
    // ... etc
};
```

Or modify the toast container position:

```html
<!-- In base.html, change this line: -->
<div id="toast-container" class="fixed bottom-4 right-4 z-50 flex flex-col gap-2"></div>

<!-- To: -->
<div id="toast-container" class="fixed top-4 right-4 z-50 flex flex-col gap-2"></div>
```

---

## Testing Toasts

Quick test in browser console:

```javascript
// Test all toast types
Toast.success('This is a success message');
Toast.error('This is an error message');
Toast.warning('This is a warning message');
Toast.info('This is an info message');

// Test custom duration
Toast.success('This will last 10 seconds', 10000);

// Test raw showToast
showToast('Testing raw function', 'success', 3000);
```

That's it! You now have a full toast notification system ready to use.
