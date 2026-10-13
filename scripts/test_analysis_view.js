(() => {
  const section = document.querySelector('.test-analysis');
  const controls = section?.querySelector('.comparison-controls');
  if (!controls) return;
  const abel = controls.querySelector('[name="abel-kernel"]');
  const external = controls.querySelector('[name="external-code"]');
  const kernels = new Set(Array.from(abel.options, o => o.value).filter(Boolean));
  const plots = Array.from(section.querySelectorAll('.analysis-plot'));
  const filtered = () => Boolean(abel.value || external.value);
  const apply = plot => {
    const doc = plot.querySelector('object')?.contentDocument;
    if (!doc?.querySelector('svg')) return;
    for (const element of doc.querySelectorAll('[id^="series--"]')) {
      const method = element.id.split('--')[1];
      const selected = kernels.has(method)
        ? !abel.value || method === abel.value
        : !external.value || method.startsWith(external.value + '_');
      element.style.opacity = selected ? '1' : '0.10';
    }
  };
  const ensure = plot => {
    if (plot.dataset.loading || !filtered()) return;
    plot.dataset.loading = 'true';
    const fallback = plot.querySelector('a');
    const object = document.createElement('object');
    object.type = 'image/svg+xml';
    object.setAttribute('role', 'img');
    object.setAttribute('aria-label', fallback.querySelector('img').alt);
    object.style.cssText = `width:100%;display:block;position:absolute;visibility:hidden;aspect-ratio:${plot.dataset.aspect}`;
    const fail = () => {
      object.remove();
      delete plot.dataset.loading;
      controls.querySelector('.comparison-status').textContent = 'Interactive emphasis is unavailable; the plots show all codes.';
    };
    object.addEventListener('error', fail, {once:true});
    object.addEventListener('load', () => {
      try {
        if (!object.contentDocument?.querySelector('svg')) return fail();
        apply(plot);
        fallback.style.display = 'none';
        object.style.position = 'static';
        object.style.visibility = 'visible';
      } catch (_) { fail(); }
    }, {once:true});
    object.data = plot.dataset.vector;
    plot.append(object);
  };
  const update = () => {
    controls.querySelector('.comparison-status').textContent = filtered()
      ? 'Selected curves are emphasized; other curves remain faint. Downloads contain all codes.'
      : 'All codes are shown on shared axes.';
    for (const plot of plots) {
      if (plot.dataset.visible === 'true') ensure(plot);
      try { apply(plot); } catch (_) { /* Static image remains available. */ }
    }
  };
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        entry.target.dataset.visible = String(entry.isIntersecting);
        if (entry.isIntersecting) ensure(entry.target);
      }
    }, {rootMargin:'400px'});
    plots.forEach(plot => observer.observe(plot));
  } else {
    plots.forEach(plot => { plot.dataset.visible = 'true'; });
  }
  abel.addEventListener('change', update);
  external.addEventListener('change', update);
  controls.querySelector('button').addEventListener('click', () => {
    abel.value = ''; external.value = ''; update();
  });
  controls.hidden = false;
})();
