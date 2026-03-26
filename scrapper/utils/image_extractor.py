import re


BAD_IMAGE_KEYWORDS = [
    "logo",
    "banner",
    "icon",
    "placeholder",
    "dealer",
    "brand",
    "marketing",
    "promo",
    "facebook",
    "instagram",
    "youtube",
    "arrow",
    "button",
    "sprite",
    "thumb-default",
    "no-image",
    "default-car",
    "svg",
    "webp?text=",
]


def normalize_url(url: str, domain: str):
    if not url:
        return None

    url = url.strip()

    if url.startswith("//"):
        return "https:" + url

    if url.startswith("/"):
        return domain.rstrip("/") + url

    if url.startswith("http://") or url.startswith("https://"):
        return url

    return None


def is_valid_car_image(url: str):
    if not url:
        return False

    lower = url.lower()

    if any(keyword in lower for keyword in BAD_IMAGE_KEYWORDS):
        return False

    if not re.search(r"\.(jpg|jpeg|png|webp)(\?.*)?$", lower):
        return False

    return True


def extract_gallery_images(soup, domain: str, max_images=20):
    candidates = []

    # anchors first = usually full-res gallery
    for a in soup.select("a[href]"):
        href = a.get("href")
        full = normalize_url(href, domain)
        if full and is_valid_car_image(full):
            candidates.append(full)

    # images second
    for img in soup.select("img"):
        src = img.get("data-src") or img.get("src")
        full = normalize_url(src, domain)
        if full and is_valid_car_image(full):
            candidates.append(full)

    seen = set()
    result = []

    for url in candidates:
        if url not in seen:
            seen.add(url)
            result.append(url)

        if len(result) >= max_images:
            break

    return result