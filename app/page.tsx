"use client";
import { useState } from "react";
import TrainPanel from "@/components/TrainPanel";
import ConvertPanel from "@/components/ConvertPanel";

const tabs = ["Train", "Convert"] as const;
type Tab = (typeof tabs)[number];
const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.VITE_API_BASE_URL ||
  "http://localhost:8000";

const Home = () => {
  const [activeTab, setActiveTab] = useState<Tab>("Train");

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse-dot" />
            <h1 className="text-lg font-mono font-bold text-foreground tracking-tight">RVC Minimal</h1>
          </div>
          <span className="text-xs font-mono text-muted-foreground">
            {apiBaseUrl}
          </span>
        </div>
      </header>

      {/* Tabs */}
      <div className="border-b border-border">
        <div className="max-w-5xl mx-auto px-6 flex gap-0">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 text-sm font-mono font-medium border-b-2 transition-colors ${
                activeTab === tab
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <main className="max-w-5xl mx-auto px-6 py-8">
        {activeTab === "Train" ? <TrainPanel /> : <ConvertPanel />}
      </main>
    </div>
  );
};

export default Home;
