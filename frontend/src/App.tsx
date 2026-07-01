import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import ClassificationPage from "./pages/ClassificationPage";
import HistoryPage from "./pages/HistoryPage";
import SettingsPage from "./pages/SettingsPage";
import { useTheme } from "./hooks/useTheme";
import { useToast } from "./hooks/useToast";
import { usePredictions } from "./hooks/usePredictions";
import type { PredictionResponse } from "./types";

export default function App() {
  const { theme, toggle } = useTheme();
  const { toasts, addToast, removeToast } = useToast();
  const { history, addPrediction, deletePrediction, clearHistory, searchHistory } =
    usePredictions();

  const handlePredictionComplete = (response: PredictionResponse, query: string) => {
    addPrediction({
      id: response.request_id,
      request: { query },
      response,
      timestamp: Date.now(),
    });
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout theme={theme} onToggleTheme={toggle} />}>
          <Route path="/" element={<HomePage />} />
          <Route
            path="/classify"
            element={
              <ClassificationPage
                onPredictionComplete={handlePredictionComplete}
                addToast={addToast}
              />
            }
          />
          <Route
            path="/history"
            element={
              <HistoryPage
                history={history}
                onDelete={deletePrediction}
                onSearch={searchHistory}
                addToast={addToast}
              />
            }
          />
          <Route
            path="/settings"
            element={
              <SettingsPage
                theme={theme}
                onToggleTheme={toggle}
                onClearHistory={clearHistory}
                addToast={addToast}
              />
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
