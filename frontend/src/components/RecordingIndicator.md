# Recording Indicator Component

## Apple-Style Recording Badge

A sleek recording indicator component inspired by Apple's design language.

## Features

✨ **Apple Design Elements:**
- Glassmorphism with backdrop blur
- SF Pro font family
- Pulsing red dot animation
- Smooth cubic-bezier transitions
- Dark/Light mode support
- Reduced motion support

🎯 **Usage:**

```jsx
import { RecordingIndicator } from './components/RecordingIndicator'

function App() {
  const [isRecording, setIsRecording] = useState(false)

  return (
    <RecordingIndicator 
      isRecording={isRecording}
      onToggle={() => setIsRecording(!isRecording)}
      showLabel={true}
    />
  )
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `isRecording` | boolean | required | Recording state |
| `onToggle` | function | required | Toggle callback |
| `showLabel` | boolean | `true` | Show "Recording" text |

## Design

**Colors:**
- Red: `#ff3b30` (Apple System Red)
- Blue: `#007aff` (Apple System Blue - focus)
- Background: Frosted glass with blur

**Animation:**
- Pulsing effect at 2s interval
- Cubic-bezier easing for smoothness
- Respects `prefers-reduced-motion`

**Typography:**
- Font: SF Pro Display (falls back to system)
- Weight: 500 (Medium) / 600 (Semibold for timer)
- Tabular numbers for timer

## Integration

### In PodcastView:

```jsx
import { RecordingIndicator } from './RecordingIndicator'

// In render:
<div className="podcast-header">
  <RecordingIndicator 
    isRecording={isPodcastRecording}
    onToggle={toggleRecording}
  />
</div>
```

### In VisionSidebar:

```jsx
<div className="vision-header">
  <h3>📹 Sign Language Input</h3>
  <RecordingIndicator 
    isRecording={isRecording}
    onToggle={onToggleRecording}
    showLabel={false} // Compact mode
  />
</div>
```

## Accessibility

- ✅ Keyboard accessible
- ✅ Focus visible outline
- ✅ Screen reader friendly labels
- ✅ Reduced motion support
- ✅ High contrast modes

## Browser Support

- ✅ Chrome/Edge 88+
- ✅ Safari 15.4+
- ✅ Firefox 103+

Requires `backdrop-filter` support for glassmorphism effect.

## Example Positions

**Top-right corner:**
```css
.recording-indicator {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
}
```

**In navbar:**
```css
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
```

## Variants

Add `.compact` class for smaller version:
```jsx
<div className="recording-indicator compact">
  {/* ... */}
</div>
```
