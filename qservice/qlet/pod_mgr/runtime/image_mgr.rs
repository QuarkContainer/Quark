// Copyright (c) 2021 Quark Container Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::BTreeMap;
use tokio::sync::Mutex as TMutex;

use qshare::common::*;
use qshare::crictl;
use qshare::k8s;
use qshare::k8s_util::*;

use crate::pod_mgr::cri::client::CriClient;

#[derive(Debug)]
pub struct ImageMgr {
    pub imageRefs: TMutex<BTreeMap<String, crictl::Image>>,
    pub imageSvc: CriClient,
    pub authConfig: crictl::AuthConfig,
}

impl ImageMgr {
    pub async fn New(authConfig: crictl::AuthConfig) -> Result<Self> {
        let imageService = CriClient::Init().await?;
        return Ok(Self {
            imageRefs: TMutex::new(BTreeMap::new()),
            imageSvc: imageService,
            authConfig: authConfig,
        });
    }

    pub async fn PullImageForContainer(
        &self,
        imageLink: &str,
        podSandboxConfig: &crictl::PodSandboxConfig,
    ) -> Result<crictl::Image> {
        let imageWithTag = match ApplyDefaultImageTag(imageLink) {
            None => {
                return Err(Error::CommonError(format!(
                    "Failed to apply default image tag {}",
                    imageLink
                )));
            }
            Some(l) => l,
        };

        match self.imageRefs.lock().await.get(&imageWithTag) {
            Some(image) => {
                info!(
                    "Container image with tag {} already present on machine",
                    &imageWithTag
                );
                return Ok(image.clone());
            }
            None => (),
        }

        let imageSpec = crictl::ImageSpec {
            image: imageWithTag.clone(),
            ..Default::default()
        };

        let filter = crictl::ImageFilter {
            image: Some(imageSpec.clone()),
        };

        let mut image = None;
        let images = self.imageSvc.ListImages(Some(filter.clone())).await?;
        for img in images {
            let tags = &img.repo_tags;
            for t in tags {
                if t.ends_with(&imageWithTag) {
                    image = Some(img);
                    break;
                }
            }
        }

        if let Some(img) = image {
            self.imageRefs
                .lock()
                .await
                .insert(imageWithTag.clone(), img.clone());
            info!(
                "Container image already present on machine image {:?} tag {}",
                &img, imageWithTag
            );
            return Ok(img);
        }

        self.imageSvc
            .PullImage(
                Some(imageSpec),
                Some(self.authConfig.clone()),
                Some(podSandboxConfig.clone()),
            )
            .await?;

        let mut image = None;
        let images = self.imageSvc.ListImages(Some(filter)).await?;
        for img in images {
            let tags = &img.repo_tags;
            for t in tags {
                if t.starts_with(&imageWithTag) {
                    image = Some(img);
                    break;
                }
            }
        }

        if let Some(img) = image {
            self.imageRefs
                .lock()
                .await
                .insert(imageWithTag.clone(), img.clone());
            return Ok(img);
        }

        panic!("impossible")
    }
}

pub fn ShouldPullImage(container: &k8s::Container, imagePresent: bool) -> bool {
    if container.image_pull_policy == Some(PullPolicy::PullNever.to_string()) {
        return false;
    }

    if container.image_pull_policy == Some(PullPolicy::PullAlways.to_string())
        || (container.image_pull_policy == Some(PullPolicy::PullIfNotPresent.to_string())
            && !imagePresent)
    {
        return true;
    }

    return false;
}

pub fn ApplyDefaultImageTag(link: &str) -> Option<String> {
    let parts: Vec<&str> = link.splitn(2, '/').collect();
    let registry;
    let remain;
    if parts.len() == 1 {
        registry = None;
        remain = parts[0];
    } else if parts.len() == 2 {
        registry = Some(parts[0]);
        remain = parts[1];
    } else {
        return None;
    }

    let repository;
    let tag;
    let parts: Vec<&str> = remain.splitn(2, ':').collect();
    if parts.len() == 1 {
        repository = parts[0];
        tag = "latest";
    } else if parts.len() == 2 {
        repository = parts[0];
        tag = parts[1];
    } else {
        return None;
    }

    if let Some(registry) = registry {
        Some(format!("{}/{}:{}", registry, repository, tag))
    } else {
        Some(format!("{}:{}", repository, tag))
    }
}

/*

pub const LEGACY_DEFAULT_DOMAIN : &str = "index.docker.io";
pub const DEFAULT_DOMAIN        : &str = "docker.io";
pub const OFFICIAL_REPO_NAME    : &str = "library";
pub const DEFAULT_TAG           : &str = "latest";

// splitDockerDomain splits a repository name to domain and remotename string.
// If no valid domain is found, the default domain is used. Repository name
// needs to be already validated before.
pub fn splitDockerDomain(name: &str) -> (String, String) {
    let domain;
    let reminder;

    if let Some(i) = name.find('/') {
        if !name.find(".:").is_none() && &name[..i] != "localhost" {
            domain = DEFAULT_DOMAIN;
            reminder = name;
        } else {
            domain = &name[..i];
            reminder = &name[i+1..];
        }
    } else {
        domain = DEFAULT_DOMAIN;
        reminder = name;
    }

    return (domain.to_string(), reminder.to_string())
}
*/
