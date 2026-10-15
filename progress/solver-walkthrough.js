(() => {
  'use strict';
  const notes = JSON.parse(document.getElementById('walkthrough-notes').textContent);
  const terms = [...document.querySelectorAll('.term')];
  const search = document.getElementById('glossary-search');
  const title = document.getElementById('note-title');
  const body = document.getElementById('note-body');
  const equations = document.getElementById('note-equations');
  const location = document.getElementById('note-location');
  const related = document.getElementById('related-terms');
  const codeLink = document.getElementById('note-code');
  let current = '';

  function filterTerms() {
    const query = search.value.trim().toLocaleLowerCase();
    let count = 0;
    terms.forEach(term => {
      term.hidden = !term.dataset.search.includes(query);
      if (!term.hidden) count++;
    });
    document.getElementById('glossary-empty').hidden = count !== 0;
  }
  function showTerm(id) {
    const term = document.getElementById(`term-${id}`);
    if (!term) return;
    search.value = '';
    filterTerms();
    term.open = true;
    term.scrollIntoView({block: 'nearest', behavior: 'instant'});
    term.querySelector('summary').focus({preventScroll: true});
  }
  function chooseNote(key) {
    const note = notes[key];
    if (!note || current === key) return;
    current = key;
    title.textContent = note.title;
    body.textContent = note.text;
    equations.innerHTML = note.equations;
    location.textContent = `L${note.lo}${note.hi !== note.lo ? `–${note.hi}` : ''}`;
    codeLink.href = `#${note.section}`;
    related.replaceChildren(...note.terms.map(id => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = document.querySelector(`#term-${id} summary`).textContent;
      button.addEventListener('click', () => showTerm(id));
      return button;
    }));
    document.querySelectorAll('.annotation').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.note === key));
    });
    document.querySelectorAll('.source-line').forEach(line => {
      const section = line.closest('.code-section');
      const n = Number(line.dataset.line);
      line.classList.toggle('selected', section.id === note.section && n >= note.lo && n <= note.hi);
    });
    // Keep the active explanation visible even after browsing lower glossary entries.
    document.querySelector('.reading-sidebar').scrollTop = 0;
  }
  document.querySelectorAll('[data-note]').forEach(element => {
    element.addEventListener('click', () => chooseNote(element.dataset.note));
    if (element.classList.contains('source-line')) {
      element.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          chooseNote(element.dataset.note);
        }
      });
    }
  });
  search.addEventListener('input', filterTerms);
  function followHash() {
    const id = decodeURIComponent(window.location.hash.slice(1));
    if (id.startsWith('term-')) showTerm(id.slice(5));
    else {
      const section = document.getElementById(id);
      if (section?.dataset.defaultNote) chooseNote(section.dataset.defaultNote);
    }
  }
  const sections = [...document.querySelectorAll('.code-section')];
  let scheduled = false;
  function trackReading() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      const active = sections.find(section => {
        const rect = section.getBoundingClientRect();
        return rect.top < window.innerHeight * .45 && rect.bottom > 60;
      });
      if (active && notes[current]?.section !== active.id) chooseNote(active.dataset.defaultNote);
    });
  }
  window.addEventListener('scroll', trackReading, {passive: true});
  window.addEventListener('hashchange', followHash);
  chooseNote('state-0');
  followHash();
  document.documentElement.classList.add('js');
})();
