#!/usr/bin/env python3
"""
Check PDF content for accuracy verification
"""

import pymupdf
import re

def check_pdf_content():
    """Extract key sections from PDF for accuracy verification"""
    try:
        # Open the PDF
        doc = pymupdf.open('documents/SRM Upgrade Guide.pdf')

        print('🔍 Extracting key sections from PDF for accuracy verification:')
        print('=' * 60)

        # Search for upgrade-related content
        upgrade_content = []
        prereq_content = []

        for page_num in range(min(20, len(doc))):  # Check first 20 pages
            page = doc[page_num]
            text = page.get_text()

            # Look for upgrade prerequisites
            if 'pre-requisites' in text.lower() or 'prerequisites' in text.lower():
                prereq_content.append(f'Page {page_num+1}: {text[:500]}...')

            # Look for upgrade steps
            if 'upgrade' in text.lower() and ('step' in text.lower() or 'procedure' in text.lower()):
                upgrade_content.append(f'Page {page_num+1}: {text[:500]}...')

        print('\n📋 Prerequisites Found:')
        for content in prereq_content[:3]:  # Show first 3
            print(f'  {content[:200]}...')

        print('\n🔧 Upgrade Steps Found:')
        for content in upgrade_content[:3]:  # Show first 3
            print(f'  {content[:200]}...')

        doc.close()

    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    check_pdf_content()
