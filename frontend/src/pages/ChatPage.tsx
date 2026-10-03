import * as React from "react";
import { LogOut, MessageSquarePlus } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";

import { ChatInput } from "@/components/chat/ChatInput";
import { MessageList } from "@/components/chat/MessageList";
import { MainLayout } from "@/components/layout/MainLayout";
import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore } from "@/stores/chatStore";

export function ChatPage() {
  const navigate = useNavigate();
  const { sessionId } = useParams<{ sessionId: string }>();
  const { user, logout } = useAuthStore();
  const {
    messages,
    isLoading,
    isStreaming,
    currentSessionId,
    sessions,
    isCreatingNew,
    fetchSessions,
    selectSession,
    createSession
  } = useChatStore();
  const showWelcome = messages.length === 0 && !isLoading;
  const [sessionsReady, setSessionsReady] = React.useState(false);
  const sessionExists = React.useMemo(() => {
    if (!sessionId) return false;
    return sessions.some((session) => session.id === sessionId);
  }, [sessionId, sessions]);

  React.useEffect(() => {
    let active = true;
    fetchSessions()
      .catch(() => null)
      .finally(() => {
        if (active) {
          setSessionsReady(true);
        }
      });
    return () => {
      active = false;
    };
  }, [fetchSessions]);

  React.useEffect(() => {
    if (sessionId) {
      if (sessionsReady && !sessionExists) {
        createSession().catch(() => null);
        navigate("/chat", { replace: true });
        return;
      }
      selectSession(sessionId).catch(() => null);
      return;
    }
    if (!sessionsReady) {
      return;
    }
    if (isCreatingNew) {
      return;
    }
    if (currentSessionId) {
      return;
    }
    createSession().catch(() => null);
  }, [
    sessionId,
    sessionsReady,
    sessionExists,
    isCreatingNew,
    currentSessionId,
    selectSession,
    createSession,
    navigate
  ]);

  React.useEffect(() => {
    if (currentSessionId && currentSessionId !== sessionId) {
      navigate(`/chat/${currentSessionId}`, { replace: true });
    }
  }, [currentSessionId, sessionId, navigate]);

  const handleNewChat = React.useCallback(() => {
    createSession().catch(() => null);
    navigate("/chat", { replace: true });
  }, [createSession, navigate]);

  const handleLogout = React.useCallback(() => {
    logout()
      .then(() => navigate("/user/login", { replace: true }))
      .catch(() => navigate("/user/login", { replace: true }));
  }, [logout, navigate]);

  if (user?.role === "admin") {
    return (
      <MainLayout>
        <div className="flex h-full flex-col bg-white">
          <div className="flex-1 min-h-0">
            <MessageList
              messages={messages}
              isLoading={isLoading}
              isStreaming={isStreaming}
              sessionKey={currentSessionId}
            />
          </div>
          {showWelcome ? null : (
            <div className="relative z-20 bg-white">
              <div className="mx-auto max-w-[800px] px-6 pt-1 pb-4">
                <ChatInput />
              </div>
            </div>
          )}
        </div>
      </MainLayout>
    );
  }

  return (
    <div className="flex h-screen min-h-screen bg-[#F7F8FA] p-0 sm:p-4">
      <section className="mx-auto flex h-full w-full max-w-[980px] flex-col overflow-hidden border border-[#E5E7EB] bg-white shadow-[0_18px_42px_rgba(15,23,42,0.08)] sm:rounded-lg">
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-[#E5E7EB] px-4">
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-[#111827]">AI 产品智能助手</p>
            <p className="truncate text-xs text-[#6B7280]">
              {user?.username || user?.userId || "用户"}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={handleNewChat}
              title="新对话"
              aria-label="新对话"
              className="h-9 w-9 rounded-lg text-[#6B7280] hover:bg-[#F3F4F6] hover:text-[#111827]"
            >
              <MessageSquarePlus className="h-4 w-4" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={handleLogout}
              title="退出登录"
              aria-label="退出登录"
              className="h-9 w-9 rounded-lg text-[#6B7280] hover:bg-[#FEF2F2] hover:text-[#DC2626]"
            >
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        </header>
        <div className="min-h-0 flex-1">
          <MessageList
            messages={messages}
            isLoading={isLoading}
            isStreaming={isStreaming}
            sessionKey={currentSessionId}
            compactEmpty
          />
        </div>
        <div className="relative z-20 shrink-0 border-t border-[#F3F4F6] bg-white">
          <div className="mx-auto max-w-[800px] px-4 py-3 sm:px-6">
            <ChatInput />
          </div>
        </div>
      </section>
    </div>
  );
}
