import * as React from "react";
import { Eye, EyeOff, Lock, MessageSquare, ShieldCheck, User } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { register as registerRequest } from "@/services/authService";
import { useAuthStore } from "@/stores/authStore";

type EntryMode = "admin" | "user";
type UserMode = "login" | "register";

export function AuthLandingPage() {
  const navigate = useNavigate();
  const { login, logout, isLoading } = useAuthStore();
  const [entryMode, setEntryMode] = React.useState<EntryMode>("user");
  const [userMode, setUserMode] = React.useState<UserMode>("login");
  const [showPassword, setShowPassword] = React.useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = React.useState(false);
  const [form, setForm] = React.useState({
    username: "",
    password: "",
    confirmPassword: ""
  });
  const [error, setError] = React.useState<string | null>(null);
  const [isRegistering, setIsRegistering] = React.useState(false);

  const isUser = entryMode === "user";
  const isRegister = isUser && userMode === "register";
  const submitting = isLoading || isRegistering;

  const resetError = () => setError(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    const username = form.username.trim();
    const password = form.password.trim();
    const confirmPassword = form.confirmPassword.trim();

    if (!username || !password) {
      setError("请输入用户名和密码。");
      return;
    }
    if (isRegister && password !== confirmPassword) {
      setError("两次输入的密码不一致。");
      return;
    }

    try {
      if (isRegister) {
        setIsRegistering(true);
        await registerRequest(username, password);
      }

      await login(username, password);
      const role = useAuthStore.getState().user?.role;

      if (entryMode === "admin" && role !== "admin") {
        await logout();
        setError("请使用管理员账号登录。");
        return;
      }

      if (entryMode === "user" && role === "admin") {
        navigate("/admin/dashboard", { replace: true });
        return;
      }

      navigate(role === "admin" ? "/admin/dashboard" : "/chat", { replace: true });
    } catch (err) {
      setError((err as Error).message || (isRegister ? "注册失败，请稍后重试。" : "登录失败，请稍后重试。"));
    } finally {
      setIsRegistering(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#F7F8FA] px-4 py-8">
      <div className="w-full max-w-[440px] rounded-lg border border-[#E5E7EB] bg-white p-6 shadow-[0_12px_30px_rgba(15,23,42,0.08)] sm:p-8">
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-[#2563EB] text-white shadow-[0_8px_18px_rgba(37,99,235,0.22)]">
            {isUser ? <MessageSquare className="h-5 w-5" /> : <ShieldCheck className="h-5 w-5" />}
          </div>
          <div>
            <p className="text-xl font-semibold text-[#111827]">AI 产品智能助手</p>
            <p className="mt-0.5 text-sm text-[#6B7280]">
              {isUser ? "用户登录后进入聊天窗口" : "管理员登录后进入系统后台"}
            </p>
          </div>
        </div>

        <div className="mb-5 grid grid-cols-2 rounded-lg bg-[#F3F4F6] p-1">
          <button
            type="button"
            onClick={() => {
              setEntryMode("admin");
              resetError();
            }}
            className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
              !isUser ? "bg-white text-[#111827] shadow-sm" : "text-[#6B7280] hover:text-[#111827]"
            }`}
          >
            管理员登录
          </button>
          <button
            type="button"
            onClick={() => {
              setEntryMode("user");
              resetError();
            }}
            className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
              isUser ? "bg-white text-[#111827] shadow-sm" : "text-[#6B7280] hover:text-[#111827]"
            }`}
          >
            用户入口
          </button>
        </div>

        {isUser ? (
          <div className="mb-5 grid grid-cols-2 rounded-lg border border-[#E5E7EB] p-1">
            <button
              type="button"
              onClick={() => {
                setUserMode("login");
                resetError();
              }}
              className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                !isRegister ? "bg-[#EFF6FF] text-[#2563EB]" : "text-[#6B7280] hover:text-[#111827]"
              }`}
            >
              登录
            </button>
            <button
              type="button"
              onClick={() => {
                setUserMode("register");
                resetError();
              }}
              className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                isRegister ? "bg-[#EFF6FF] text-[#2563EB]" : "text-[#6B7280] hover:text-[#111827]"
              }`}
            >
              注册
            </button>
          </div>
        ) : null}

        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <label className="text-sm font-medium text-[#374151]">用户名</label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#9CA3AF]" />
              <Input
                placeholder={isUser ? "请输入用户名" : "请输入管理员用户名"}
                value={form.username}
                onChange={(event) => setForm((prev) => ({ ...prev, username: event.target.value }))}
                className="h-11 rounded-lg border-[#D1D5DB] bg-white pl-10 shadow-none focus-visible:ring-[#93C5FD]"
                autoComplete="username"
              />
            </div>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-[#374151]">密码</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#9CA3AF]" />
              <Input
                type={showPassword ? "text" : "password"}
                placeholder="请输入密码"
                value={form.password}
                onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value }))}
                className="h-11 rounded-lg border-[#D1D5DB] bg-white pl-10 pr-10 shadow-none focus-visible:ring-[#93C5FD]"
                autoComplete={isRegister ? "new-password" : "current-password"}
              />
              <button
                type="button"
                onClick={() => setShowPassword((prev) => !prev)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-[#9CA3AF] transition-colors hover:text-[#4B5563]"
                aria-label="显示或隐藏密码"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>
          {isRegister ? (
            <div className="space-y-2">
              <label className="text-sm font-medium text-[#374151]">确认密码</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#9CA3AF]" />
                <Input
                  type={showConfirmPassword ? "text" : "password"}
                  placeholder="请再次输入密码"
                  value={form.confirmPassword}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, confirmPassword: event.target.value }))
                  }
                  className="h-11 rounded-lg border-[#D1D5DB] bg-white pl-10 pr-10 shadow-none focus-visible:ring-[#93C5FD]"
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword((prev) => !prev)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[#9CA3AF] transition-colors hover:text-[#4B5563]"
                  aria-label="显示或隐藏确认密码"
                >
                  {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>
          ) : null}
          {error ? <p className="text-sm text-[#DC2626]">{error}</p> : null}
          <Button
            type="submit"
            className="h-11 w-full rounded-lg bg-[#2563EB] shadow-none hover:bg-[#1D4ED8]"
            disabled={submitting}
          >
            {submitting
              ? isRegister
                ? "正在注册..."
                : "正在登录..."
              : isRegister
                ? "注册并登录"
                : "登录"}
          </Button>
        </form>

        <div className="mt-5 flex justify-center gap-4 text-xs text-[#6B7280]">
          <Link to="/login" className="font-medium text-[#2563EB] hover:underline">
            管理员直达
          </Link>
          <Link to="/user/login" className="font-medium text-[#2563EB] hover:underline">
            用户直达
          </Link>
        </div>
      </div>
    </div>
  );
}
