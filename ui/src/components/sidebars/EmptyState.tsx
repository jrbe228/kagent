"use client";
import { MessageSquare, Users } from "lucide-react";
import { useRouter } from "next/navigation";
import KagentLogo from "../kagent-logo";
import { SidebarMenuButton } from "../ui/sidebar";

const EmptyState = () => (
  <div className="h-full flex flex-col items-center justify-center p-8 text-center">
    <div className="bg-primary rounded-full p-4 mb-4">
      <MessageSquare className="h-8 w-8 text-primary-foreground " />
    </div>
    <h3 className="text-lg font-semibold mb-2">No chats yet</h3>
    <p className="text-sm  max-w-[250px] mb-6">{"Start a new conversation to begin using kagent"}</p>
    <ActionButtons hasSessions={false} />
  </div>
);

interface ActionButtonsProps {
  hasSessions?: boolean;
  currentAgentId?: number;
}
const ActionButtons = ({ hasSessions, currentAgentId }: ActionButtonsProps) => {
  const router = useRouter();

  return (
    <div className="px-2 space-y-4">
      {hasSessions && currentAgentId && (
        <SidebarMenuButton onClick={() => router.push(`/agents/${currentAgentId}/chat`)}>
          <KagentLogo className="mr-3 h-4 w-4" />
          <span>Start a new chat</span>
        </SidebarMenuButton>
      )}
      <SidebarMenuButton onClick={() => router.push(`/`)}>
        <Users className="mr-3 h-4 w-4" />
        <span>Switch Agent</span>
        </SidebarMenuButton>
    </div>
  );
};
export { EmptyState, ActionButtons };
