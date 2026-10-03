import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

// Bootstrap supplies the grid, forms and utilities; the portal stylesheet layers the Jobick
// visual language on top, so it must be imported second.
import 'bootstrap/dist/css/bootstrap.min.css';
import './styles/jobick.css';

import { App } from './App';

const container = document.getElementById('root');

if (!container) {
  throw new Error('The root element is missing from index.html.');
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
