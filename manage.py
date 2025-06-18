#!/usr/bin/env python
import os
import sys

def main():
    """Jalankan perintah manajemen Django."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'quail_project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Tidak dapat mengimpor Django. Pastikan Django sudah diinstal dan tersedia di lingkungan Python kamu."
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
