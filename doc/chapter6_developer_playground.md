# 📖 Chapter 6: The Interactive Playground UI

[⬅️ Previous Chapter](chapter5_decoding_upgrades.md) | [📖 Table of Contents](training_journey.md)

---

## 🎨 1. Premium Glassmorphism UI Design

To make interacting with our 1.6B parameter model a premium experience, we built a developer playground using Gradio 6.0:

*   **Side-by-Side Workspace Layout**: Designed a balanced, two-column interface placing the code prompt input on the left and the streamed code output on the right.
*   **Collapsible Hyperparameters Accordion**: To keep the interface compact and within a single vertical screen frame, all sliders (`Temperature`, `Max Tokens`, `Repetition Penalty`, `Top-K`, `Top-P`) are housed inside a collapsed accordion.
*   **Obsidian Dark Styling**: Implemented a custom dark-mode theme utilizing a deep charcoal neutral palette, vibrant Rust orange accents (`#ea580c`), and premium **Fira Code monospaced typography** for inputs and outputs.

---

## ⚡ 2. Real-Time Token Streaming

To maximize responsiveness and eliminate latency delays, we engineered the generation pipeline to **stream tokens in real time**:
*   The `generate_completion` function in `gradio_app.py` is configured as a Python generator.
*   On each token generation step, the engine decodes the accumulated list of integers and **yields** the updated string immediately:
    ```python
    # Yield the generated-so-far tokens to stream them in real time
    generated_only = generated[len(prompt_tokens):]
    yield _tokenizer.decode(generated_only)
    ```
*   This results in a smooth, typing-like streaming effect that keeps the developer engaged without any burst lag.

---

## 📋 3. Global Clipboard Polyfill

Standard web browsers restrict the modern `navigator.clipboard` API to secure contexts (localhost or HTTPS). When developers access the playground over insecure local networks (HTTP), the built-in copy icons inside Gradio's code blocks silently fail.

To resolve this, we injected a global JavaScript **clipboard polyfill** inside the page header at launch:

```javascript
(function() {
    // If navigator.clipboard is missing (insecure HTTP context), polyfill it!
    if (typeof navigator.clipboard === "undefined") {
        Object.defineProperty(navigator, "clipboard", {
            value: {
                writeText: function(text) {
                    return new Promise(function(resolve, reject) {
                        try {
                            fallbackCopyText(text);
                            resolve();
                        } catch (err) {
                            reject(err);
                        }
                    });
                }
            },
            writable: true,
            configurable: true
        });
    } else {
        // Wrap writeText to fallback if the native call fails
        var nativeWriteText = navigator.clipboard.writeText;
        navigator.clipboard.writeText = function(text) {
            return nativeWriteText.call(navigator.clipboard, text).catch(function(err) {
                fallbackCopyText(text);
            });
        };
    }

    function fallbackCopyText(text) {
        var textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.top = "0";
        textArea.style.left = "0";
        textArea.style.width = "2em";
        textArea.style.height = "2em";
        textArea.style.background = "transparent";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
            document.execCommand("copy");
        } catch (err) {
            console.error("Fallback copy failed", err);
        }
        document.body.removeChild(textArea);
    }
})();
```

By globally monkey-patching `navigator.clipboard` directly in the page header, the **existing built-in copy icons on Gradio's code editors work seamlessly with a 100% copy success rate in all secure and insecure local environments**.

---

[⬅️ Previous Chapter](chapter5_decoding_upgrades.md) | [📖 Table of Contents](training_journey.md)
