module.exports = {
  apps: [
    {
      name: 'ai-service',

      // Run gunicorn directly — no pm2-python needed
      script: '/home/ubuntu/ai-service/venv/bin/gunicorn',
      args: [
        'ai_study_project.wsgi:application',
        '--bind', '0.0.0.0:8000',
        '--workers', '3',
        '--timeout', '120',
        '--access-logfile', '-',
        '--error-logfile', '-',
        '--log-level', 'info',
      ].join(' '),
      interpreter: 'none',
      cwd: '/home/ubuntu/ai-service',

      autorestart: true,
      watch: false,
      max_memory_restart: '1G',

      env: {
        DJANGO_DEBUG: 'false',
        ALLOWED_HOSTS: '0.0.0.0,127.0.0.1,localhost',
      },

      out_file: '/home/ubuntu/logs/ai-out.log',
      error_file: '/home/ubuntu/logs/ai-err.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,

      kill_timeout: 10000,
    },
  ],
};
