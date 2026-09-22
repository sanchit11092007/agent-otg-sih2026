import AgenticBrowser from './AgenticBrowser';
import ReceiverPage from './ReceiverPage';
import './index.css'; // Make sure your Tailwind directives are in here

export default function App() {
  // /receiver is the intentionally small page shared with a second PC.
  if (window.location.pathname.replace(/\/+$/, '') === '/receiver') {
    return <ReceiverPage />;
  }

  return (
    <AgenticBrowser />
  );
}
