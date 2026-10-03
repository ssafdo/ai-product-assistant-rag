-- PostgreSQL Initial Data for AI Product Assistant RAG

INSERT INTO t_user (id, username, password, role, avatar, create_time, update_time, deleted)
VALUES (2001523723396308993, 'admin', 'admin', 'admin', 'https://static.deepseek.com/user-avatar/G_6cuD8GbD53VwGRwisvCsZ6', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0);

INSERT INTO t_sample_question (id, title, description, question, create_time, update_time, deleted)
VALUES
  (2100000000000000001, '功能解读', '说明产品能力、边界和适用场景', '请说明 AI 产品智能助手的核心能力、适用场景和当前限制。', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0),
  (2100000000000000002, '套餐对比', '比较版本权益和推荐客户', '请对比基础版、专业版和企业版的核心差异，并说明分别适合什么客户。', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0),
  (2100000000000000003, '售前话术', '生成面向客户的沟通口径', '客户问为什么不用通用大模型直接回答，请给出一版售前回应话术。', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0);
