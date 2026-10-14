import { createRoot } from 'react-dom/client'
import '@fontsource/public-sans/400.css'
import '@fontsource/public-sans/500.css'
import '@fontsource/public-sans/600.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import '@fontsource-variable/bricolage-grotesque'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  // No StrictMode: its dev double-mount leaves drei <Html> labels empty.
  <App />,
)
